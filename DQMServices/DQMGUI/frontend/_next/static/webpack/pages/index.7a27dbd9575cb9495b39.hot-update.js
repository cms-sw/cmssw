webpackHotUpdate_N_E("pages/index",{

/***/ "./components/Nav.tsx":
/*!****************************!*\
  !*** ./components/Nav.tsx ***!
  \****************************/
/*! exports provided: Nav, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Nav", function() { return Nav; });
/* harmony import */ var _babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/extends */ "./node_modules/@babel/runtime/helpers/esm/extends.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _searchButton__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./searchButton */ "./components/searchButton.tsx");
/* harmony import */ var _helpButton__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./helpButton */ "./components/helpButton.tsx");



var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/Nav.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_2___default.a.createElement;





var Nav = function Nav(_ref) {
  _s();

  var initial_search_run_number = _ref.initial_search_run_number,
      initial_search_dataset_name = _ref.initial_search_dataset_name,
      setRunNumber = _ref.setRunNumber,
      setDatasetName = _ref.setDatasetName,
      handler = _ref.handler,
      type = _ref.type,
      defaultRunNumber = _ref.defaultRunNumber,
      defaultDatasetName = _ref.defaultDatasetName;

  var _Form$useForm = antd__WEBPACK_IMPORTED_MODULE_3__["Form"].useForm(),
      _Form$useForm2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_Form$useForm, 1),
      form = _Form$useForm2[0];

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initial_search_run_number || ''),
      form_search_run_number = _useState[0],
      setFormRunNumber = _useState[1];

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initial_search_dataset_name || ''),
      form_search_dataset_name = _useState2[0],
      setFormDatasetName = _useState2[1]; // We have to wait for changin initial_search_run_number and initial_search_dataset_name coming from query, because the first render they are undefined and therefore the initialValues doesn't grab them


  Object(react__WEBPACK_IMPORTED_MODULE_2__["useEffect"])(function () {
    form.resetFields();
    setFormRunNumber(initial_search_run_number || '');
    setFormDatasetName(initial_search_dataset_name || '');
  }, [initial_search_run_number, initial_search_dataset_name, form]);
  var layout = {
    labelCol: {
      span: 8
    },
    wrapperCol: {
      span: 16
    }
  };
  var tailLayout = {
    wrapperCol: {
      offset: 0,
      span: 4
    }
  };
  return __jsx("div", {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 55,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["CustomForm"], Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({
    form: form,
    layout: 'inline',
    justifycontent: "center",
    width: "max-content"
  }, layout, {
    name: "search_form".concat(type),
    className: "fieldLabel",
    initialValues: {
      run_number: initial_search_run_number,
      dataset_name: initial_search_dataset_name
    },
    onFinish: function onFinish() {
      setRunNumber && setRunNumber(form_search_run_number);
      setDatasetName && setDatasetName(form_search_dataset_name);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 56,
      columnNumber: 7
    }
  }), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({}, tailLayout, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 73,
      columnNumber: 9
    }
  }), __jsx(_helpButton__WEBPACK_IMPORTED_MODULE_6__["QuestionButton"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 74,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "run_number",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 76,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "run_number",
    onChange: function onChange(e) {
      return setFormRunNumber(e.target.value);
    },
    placeholder: "Enter run number",
    type: "text",
    name: "run_number",
    value: defaultRunNumber,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 77,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "dataset_name",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 88,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "dataset_name",
    placeholder: "Enter dataset name",
    onChange: function onChange(e) {
      return setFormDatasetName(e.target.value);
    },
    type: "text",
    value: defaultDatasetName,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 89,
      columnNumber: 11
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({}, tailLayout, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 99,
      columnNumber: 9
    }
  }), __jsx(_searchButton__WEBPACK_IMPORTED_MODULE_5__["SearchButton"], {
    onClick: function onClick() {
      return handler(form_search_run_number, form_search_dataset_name, form_search_lumisection);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 100,
      columnNumber: 11
    }
  }))));
};

_s(Nav, "d/o1hn25bH6EF0LAvbTEx8d/DOY=", false, function () {
  return [antd__WEBPACK_IMPORTED_MODULE_3__["Form"].useForm];
});

_c = Nav;
/* harmony default export */ __webpack_exports__["default"] = (Nav);

var _c;

$RefreshReg$(_c, "Nav");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9OYXYudHN4Il0sIm5hbWVzIjpbIk5hdiIsImluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIiLCJpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUiLCJzZXRSdW5OdW1iZXIiLCJzZXREYXRhc2V0TmFtZSIsImhhbmRsZXIiLCJ0eXBlIiwiZGVmYXVsdFJ1bk51bWJlciIsImRlZmF1bHREYXRhc2V0TmFtZSIsIkZvcm0iLCJ1c2VGb3JtIiwiZm9ybSIsInVzZVN0YXRlIiwiZm9ybV9zZWFyY2hfcnVuX251bWJlciIsInNldEZvcm1SdW5OdW1iZXIiLCJmb3JtX3NlYXJjaF9kYXRhc2V0X25hbWUiLCJzZXRGb3JtRGF0YXNldE5hbWUiLCJ1c2VFZmZlY3QiLCJyZXNldEZpZWxkcyIsImxheW91dCIsImxhYmVsQ29sIiwic3BhbiIsIndyYXBwZXJDb2wiLCJ0YWlsTGF5b3V0Iiwib2Zmc2V0IiwicnVuX251bWJlciIsImRhdGFzZXRfbmFtZSIsImUiLCJ0YXJnZXQiLCJ2YWx1ZSIsImZvcm1fc2VhcmNoX2x1bWlzZWN0aW9uIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQUE7QUFDQTtBQUVBO0FBQ0E7QUFDQTtBQWVPLElBQU1BLEdBQUcsR0FBRyxTQUFOQSxHQUFNLE9BU0g7QUFBQTs7QUFBQSxNQVJkQyx5QkFRYyxRQVJkQSx5QkFRYztBQUFBLE1BUGRDLDJCQU9jLFFBUGRBLDJCQU9jO0FBQUEsTUFOZEMsWUFNYyxRQU5kQSxZQU1jO0FBQUEsTUFMZEMsY0FLYyxRQUxkQSxjQUtjO0FBQUEsTUFKZEMsT0FJYyxRQUpkQSxPQUljO0FBQUEsTUFIZEMsSUFHYyxRQUhkQSxJQUdjO0FBQUEsTUFGZEMsZ0JBRWMsUUFGZEEsZ0JBRWM7QUFBQSxNQURkQyxrQkFDYyxRQURkQSxrQkFDYzs7QUFBQSxzQkFDQ0MseUNBQUksQ0FBQ0MsT0FBTCxFQUREO0FBQUE7QUFBQSxNQUNQQyxJ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LHlDQUFELENBQU0sSUFBTix5RkFBZWUsVUFBZjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLE1BQ0UsTUFBQywwREFBRDtBQUNFLFdBQU8sRUFBRTtBQUFBLGFBQ1BsQixPQUFPLENBQUNRLHNCQUFELEVBQXlCRSx3QkFBekIsRUFBbURlLHVCQUFuRCxDQURBO0FBQUEsS0FEWDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0EzQ0YsQ0FERixDQURGO0FBdURELENBeEZNOztHQUFNOUIsRztVQVVJUyx5Q0FBSSxDQUFDQyxPOzs7S0FWVFYsRztBQTBGRUEsa0VBQWYiLCJmaWxlIjoic3RhdGljL3dlYnBhY2svcGFnZXMvaW5kZXguN2EyN2RiZDk1NzVjYjk0OTViMzkuaG90LXVwZGF0ZS5qcyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCBSZWFjdCwgeyBDaGFuZ2VFdmVudCwgRGlzcGF0Y2gsIHVzZUVmZmVjdCwgdXNlU3RhdGUgfSBmcm9tICdyZWFjdCc7XG5pbXBvcnQgeyBGb3JtIH0gZnJvbSAnYW50ZCc7XG5cbmltcG9ydCB7IFN0eWxlZEZvcm1JdGVtLCBTdHlsZWRJbnB1dCwgQ3VzdG9tRm9ybSB9IGZyb20gJy4vc3R5bGVkQ29tcG9uZW50cyc7XG5pbXBvcnQgeyBTZWFyY2hCdXR0b24gfSBmcm9tICcuL3NlYXJjaEJ1dHRvbic7XG5pbXBvcnQgeyBRdWVzdGlvbkJ1dHRvbiB9IGZyb20gJy4vaGVscEJ1dHRvbic7XG5pbXBvcnQgeyBmdW5jdGlvbnNfY29uZmlnIH0gZnJvbSAnLi4vY29uZmlnL2NvbmZpZyc7XG5cbmludGVyZmFjZSBOYXZQcm9wcyB7XG4gIHNldFJ1bk51bWJlcj86IERpc3BhdGNoPGFueT47XG4gIHNldERhdGFzZXROYW1lPzogRGlzcGF0Y2g8YW55PjtcbiAgaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlcj86IHN0cmluZztcbiAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lPzogc3RyaW5nO1xuICBpbml0aWFsX3NlYXJjaF9sdW1pc2VjdGlvbj86IHN0cmluZztcbiAgaGFuZGxlcihzZWFyY2hfYnlfcnVuX251bWJlcjogc3RyaW5nLCBzZWFyY2hfYnlfZGF0YXNldF9uYW1lOiBzdHJpbmcsIHNlYXJjaF9ieV9sdW1pc2VjdGlvbjogc3RyaW5nKTogdm9pZDtcbiAgdHlwZTogc3RyaW5nO1xuICBkZWZhdWx0UnVuTnVtYmVyPzogdW5kZWZpbmVkIHwgc3RyaW5nO1xuICBkZWZhdWx0RGF0YXNldE5hbWU/OiBzdHJpbmcgfCB1bmRlZmluZWQ7XG59XG5cbmV4cG9ydCBjb25zdCBOYXYgPSAoe1xuICBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyLFxuICBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUsXG4gIHNldFJ1bk51bWJlcixcbiAgc2V0RGF0YXNldE5hbWUsXG4gIGhhbmRsZXIsXG4gIHR5cGUsXG4gIGRlZmF1bHRSdW5OdW1iZXIsXG4gIGRlZmF1bHREYXRhc2V0TmFtZSxcbn06IE5hdlByb3BzKSA9PiB7XG4gIGNvbnN0IFtmb3JtXSA9IEZvcm0udXNlRm9ybSgpO1xuICBjb25zdCBbZm9ybV9zZWFyY2hfcnVuX251bWJlciwgc2V0Rm9ybVJ1bk51bWJlcl0gPSB1c2VTdGF0ZShcbiAgICBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyIHx8ICcnXG4gICk7XG4gIGNvbnN0IFtmb3JtX3NlYXJjaF9kYXRhc2V0X25hbWUsIHNldEZvcm1EYXRhc2V0TmFtZV0gPSB1c2VTdGF0ZShcbiAgICBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUgfHwgJydcbiAgKTtcblxuICAvLyBXZSBoYXZlIHRvIHdhaXQgZm9yIGNoYW5naW4gaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlciBhbmQgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lIGNvbWluZyBmcm9tIHF1ZXJ5LCBiZWNhdXNlIHRoZSBmaXJzdCByZW5kZXIgdGhleSBhcmUgdW5kZWZpbmVkIGFuZCB0aGVyZWZvcmUgdGhlIGluaXRpYWxWYWx1ZXMgZG9lc24ndCBncmFiIHRoZW1cbiAgdXNlRWZmZWN0KCgpID0+IHtcbiAgICBmb3JtLnJlc2V0RmllbGRzKCk7XG4gICAgc2V0Rm9ybVJ1bk51bWJlcihpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyIHx8ICcnKTtcbiAgICBzZXRGb3JtRGF0YXNldE5hbWUoaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lIHx8ICcnKTtcbiAgfSwgW2luaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIsIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSxmb3JtXSk7XG5cbiAgY29uc3QgbGF5b3V0ID0ge1xuICAgIGxhYmVsQ29sOiB7IHNwYW46IDggfSxcbiAgICB3cmFwcGVyQ29sOiB7IHNwYW46IDE2IH0sXG4gIH07XG4gIGNvbnN0IHRhaWxMYXlvdXQgPSB7XG4gICAgd3JhcHBlckNvbDogeyBvZmZzZXQ6IDAsIHNwYW46IDQgfSxcbiAgfTtcblxuICByZXR1cm4gKFxuICAgIDxkaXY+XG4gICAgICA8Q3VzdG9tRm9ybVxuICAgICAgICBmb3JtPXtmb3JtfVxuICAgICAgICBsYXlvdXQ9eydpbmxpbmUnfVxuICAgICAgICBqdXN0aWZ5Y29udGVudD1cImNlbnRlclwiXG4gICAgICAgIHdpZHRoPVwibWF4LWNvbnRlbnRcIlxuICAgICAgICB7Li4ubGF5b3V0fVxuICAgICAgICBuYW1lPXtgc2VhcmNoX2Zvcm0ke3R5cGV9YH1cbiAgICAgICAgY2xhc3NOYW1lPVwiZmllbGRMYWJlbFwiXG4gICAgICAgIGluaXRpYWxWYWx1ZXM9e3tcbiAgICAgICAgICBydW5fbnVtYmVyOiBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyLFxuICAgICAgICAgIGRhdGFzZXRfbmFtZTogaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lLFxuICAgICAgICB9fVxuICAgICAgICBvbkZpbmlzaD17KCkgPT4ge1xuICAgICAgICAgIHNldFJ1bk51bWJlciAmJiBzZXRSdW5OdW1iZXIoZm9ybV9zZWFyY2hfcnVuX251bWJlcik7XG4gICAgICAgICAgc2V0RGF0YXNldE5hbWUgJiYgc2V0RGF0YXNldE5hbWUoZm9ybV9zZWFyY2hfZGF0YXNldF9uYW1lKTtcbiAgICAgICAgfX1cbiAgICAgID5cbiAgICAgICAgPEZvcm0uSXRlbSB7Li4udGFpbExheW91dH0+XG4gICAgICAgICAgPFF1ZXN0aW9uQnV0dG9uIC8+XG4gICAgICAgIDwvRm9ybS5JdGVtPlxuICAgICAgICA8U3R5bGVkRm9ybUl0ZW0gbmFtZT1cInJ1bl9udW1iZXJcIj5cbiAgICAgICAgICA8U3R5bGVkSW5wdXRcbiAgICAgICAgICAgIGlkPVwicnVuX251bWJlclwiXG4gICAgICAgICAgICBvbkNoYW5nZT17KGU6IENoYW5nZUV2ZW50PEhUTUxJbnB1dEVsZW1lbnQ+KSA9PlxuICAgICAgICAgICAgICBzZXRGb3JtUnVuTnVtYmVyKGUudGFyZ2V0LnZhbHVlKVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgcGxhY2Vob2xkZXI9XCJFbnRlciBydW4gbnVtYmVyXCJcbiAgICAgICAgICAgIHR5cGU9XCJ0ZXh0XCJcbiAgICAgICAgICAgIG5hbWU9XCJydW5fbnVtYmVyXCJcbiAgICAgICAgICAgIHZhbHVlPXtkZWZhdWx0UnVuTnVtYmVyfVxuICAgICAgICAgIC8+XG4gICAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XG4gICAgICAgIDxTdHlsZWRGb3JtSXRlbSBuYW1lPVwiZGF0YXNldF9uYW1lXCI+XG4gICAgICAgICAgPFN0eWxlZElucHV0XG4gICAgICAgICAgICBpZD1cImRhdGFzZXRfbmFtZVwiXG4gICAgICAgICAgICBwbGFjZWhvbGRlcj1cIkVudGVyIGRhdGFzZXQgbmFtZVwiXG4gICAgICAgICAgICBvbkNoYW5nZT17KGU6IENoYW5nZUV2ZW50PEhUTUxJbnB1dEVsZW1lbnQ+KSA9PlxuICAgICAgICAgICAgICBzZXRGb3JtRGF0YXNldE5hbWUoZS50YXJnZXQudmFsdWUpXG4gICAgICAgICAgICB9XG4gICAgICAgICAgICB0eXBlPVwidGV4dFwiXG4gICAgICAgICAgICB2YWx1ZT17ZGVmYXVsdERhdGFzZXROYW1lfVxuICAgICAgICAgIC8+XG4gICAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XG4gICAgICAgIDxGb3JtLkl0ZW0gey4uLnRhaWxMYXlvdXR9PlxuICAgICAgICAgIDxTZWFyY2hCdXR0b25cbiAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+XG4gICAgICAgICAgICAgIGhhbmRsZXIoZm9ybV9zZWFyY2hfcnVuX251bWJlciwgZm9ybV9zZWFyY2hfZGF0YXNldF9uYW1lLCBmb3JtX3NlYXJjaF9sdW1pc2VjdGlvbilcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAvPlxuICAgICAgICA8L0Zvcm0uSXRlbT5cbiAgICAgIDwvQ3VzdG9tRm9ybT5cbiAgICA8L2Rpdj5cbiAgKTtcbn07XG5cbmV4cG9ydCBkZWZhdWx0IE5hdjtcbiJdLCJzb3VyY2VSb290IjoiIn0=