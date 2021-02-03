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
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../config/config */ "./config/config.ts");



var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/Nav.tsx",
    _this = undefined,
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
  return __jsx("div", {
    style: {
      justifyContent: 'center',
      width: 'max-content'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 52,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["CustomForm"], Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({
    form: form,
    layout: 'inline',
    justifycontent: "center"
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
      lineNumber: 53,
      columnNumber: 7
    }
  }), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 69,
      columnNumber: 9
    }
  }, __jsx(_helpButton__WEBPACK_IMPORTED_MODULE_6__["QuestionButton"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 70,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "run_number",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 72,
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
      lineNumber: 73,
      columnNumber: 11
    }
  })), _config_config__WEBPACK_IMPORTED_MODULE_7__["functions_config"].mode !== 'ONLINE' && __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "dataset_name",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 85,
      columnNumber: 11
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
      lineNumber: 86,
      columnNumber: 13
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 97,
      columnNumber: 9
    }
  }, __jsx(_searchButton__WEBPACK_IMPORTED_MODULE_5__["SearchButton"], {
    onClick: function onClick() {
      return handler(form_search_run_number, form_search_dataset_name);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 98,
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9OYXYudHN4Il0sIm5hbWVzIjpbIk5hdiIsImluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIiLCJpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUiLCJzZXRSdW5OdW1iZXIiLCJzZXREYXRhc2V0TmFtZSIsImhhbmRsZXIiLCJ0eXBlIiwiZGVmYXVsdFJ1bk51bWJlciIsImRlZmF1bHREYXRhc2V0TmFtZSIsIkZvcm0iLCJ1c2VGb3JtIiwiZm9ybSIsInVzZVN0YXRlIiwiZm9ybV9zZWFyY2hfcnVuX251bWJlciIsInNldEZvcm1SdW5OdW1iZXIiLCJmb3JtX3NlYXJjaF9kYXRhc2V0X25hbWUiLCJzZXRGb3JtRGF0YXNldE5hbWUiLCJ1c2VFZmZlY3QiLCJyZXNldEZpZWxkcyIsImxheW91dCIsImxhYmVsQ29sIiwic3BhbiIsIndyYXBwZXJDb2wiLCJqdXN0aWZ5Q29udGVudCIsIndpZHRoIiwicnVuX251bWJlciIsImRhdGFzZXRfbmFtZSIsImUiLCJ0YXJnZXQiLCJ2YWx1ZSIsImZ1bmN0aW9uc19jb25maWciLCJtb2RlIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQWNPLElBQU1BLEdBQUcsR0FBRyxTQUFOQSxHQUFNLE9BU0g7QUFBQTs7QUFBQSxNQVJkQyx5QkFRYyxRQVJkQSx5QkFRYztBQUFBLE1BUGRDLDJCQU9jLFFBUGRBLDJCQU9jO0FBQUEsTUFOZEMsWUFNYyxRQU5kQSxZQU1jO0FBQUEsTUFMZEMsY0FLYyxRQUxkQSxjQUtjO0FBQUEsTUFKZEMsT0FJYyxRQUpkQSxPQUljO0FBQUEsTUFIZEMsSUFHYyxRQUhkQSxJQUdjO0FBQUEsTUFGZEMsZ0JBRWMsUUFGZEEsZ0JBRWM7QUFBQSxNQURkQyxrQkFDYyxRQURkQSxrQkFDYzs7QUFBQSxzQkFDQ0MseUNBQUksQ0FBQ0MsT0FBTCxFQUREO0FBQUE7QUFB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SUFBTjtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLEtBQ0UsTUFBQywwREFBRDtBQUNFLFdBQU8sRUFBRTtBQUFBLGFBQ1BILE9BQU8sQ0FBQ1Esc0JBQUQsRUFBeUJFLHdCQUF6QixDQURBO0FBQUEsS0FEWDtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBLElBREYsQ0E1Q0YsQ0FERixDQURGO0FBd0RELENBdEZNOztHQUFNZixHO1VBVUlTLHlDQUFJLENBQUNDLE87OztLQVZUVixHO0FBd0ZFQSxrRUFBZiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC5lYjI0NTgzMjQ4OWE0ZTZmZDcwMy5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IENoYW5nZUV2ZW50LCBEaXNwYXRjaCwgdXNlRWZmZWN0LCB1c2VTdGF0ZSB9IGZyb20gJ3JlYWN0JztcclxuaW1wb3J0IHsgRm9ybSB9IGZyb20gJ2FudGQnO1xyXG5cclxuaW1wb3J0IHsgU3R5bGVkRm9ybUl0ZW0sIFN0eWxlZElucHV0LCBDdXN0b21Gb3JtIH0gZnJvbSAnLi9zdHlsZWRDb21wb25lbnRzJztcclxuaW1wb3J0IHsgU2VhcmNoQnV0dG9uIH0gZnJvbSAnLi9zZWFyY2hCdXR0b24nO1xyXG5pbXBvcnQgeyBRdWVzdGlvbkJ1dHRvbiB9IGZyb20gJy4vaGVscEJ1dHRvbic7XHJcbmltcG9ydCB7IGZ1bmN0aW9uc19jb25maWcgfSBmcm9tICcuLi9jb25maWcvY29uZmlnJztcclxuXHJcbmludGVyZmFjZSBOYXZQcm9wcyB7XHJcbiAgc2V0UnVuTnVtYmVyPzogRGlzcGF0Y2g8YW55PjtcclxuICBzZXREYXRhc2V0TmFtZT86IERpc3BhdGNoPGFueT47XHJcbiAgaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlcj86IHN0cmluZztcclxuICBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWU/OiBzdHJpbmc7XHJcbiAgaW5pdGlhbF9zZWFyY2hfbHVtaXNlY3Rpb24/OiBzdHJpbmc7XHJcbiAgaGFuZGxlcihzZWFyY2hfYnlfcnVuX251bWJlcjogc3RyaW5nLCBzZWFyY2hfYnlfZGF0YXNldF9uYW1lOiBzdHJpbmcpOiB2b2lkO1xyXG4gIHR5cGU6IHN0cmluZztcclxuICBkZWZhdWx0UnVuTnVtYmVyPzogdW5kZWZpbmVkIHwgc3RyaW5nO1xyXG4gIGRlZmF1bHREYXRhc2V0TmFtZT86IHN0cmluZyB8IHVuZGVmaW5lZDtcclxufVxyXG5cclxuZXhwb3J0IGNvbnN0IE5hdiA9ICh7XHJcbiAgaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlcixcclxuICBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUsXHJcbiAgc2V0UnVuTnVtYmVyLFxyXG4gIHNldERhdGFzZXROYW1lLFxyXG4gIGhhbmRsZXIsXHJcbiAgdHlwZSxcclxuICBkZWZhdWx0UnVuTnVtYmVyLFxyXG4gIGRlZmF1bHREYXRhc2V0TmFtZSxcclxufTogTmF2UHJvcHMpID0+IHtcclxuICBjb25zdCBbZm9ybV0gPSBGb3JtLnVzZUZvcm0oKTtcclxuICBjb25zdCBbZm9ybV9zZWFyY2hfcnVuX251bWJlciwgc2V0Rm9ybVJ1bk51bWJlcl0gPSB1c2VTdGF0ZShcclxuICAgIGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIgfHwgJydcclxuICApO1xyXG4gIGNvbnN0IFtmb3JtX3NlYXJjaF9kYXRhc2V0X25hbWUsIHNldEZvcm1EYXRhc2V0TmFtZV0gPSB1c2VTdGF0ZShcclxuICAgIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSB8fCAnJ1xyXG4gICk7XHJcblxyXG4gIC8vIFdlIGhhdmUgdG8gd2FpdCBmb3IgY2hhbmdpbiBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyIGFuZCBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUgY29taW5nIGZyb20gcXVlcnksIGJlY2F1c2UgdGhlIGZpcnN0IHJlbmRlciB0aGV5IGFyZSB1bmRlZmluZWQgYW5kIHRoZXJlZm9yZSB0aGUgaW5pdGlhbFZhbHVlcyBkb2Vzbid0IGdyYWIgdGhlbVxyXG4gIHVzZUVmZmVjdCgoKSA9PiB7XHJcbiAgICBmb3JtLnJlc2V0RmllbGRzKCk7XHJcbiAgICBzZXRGb3JtUnVuTnVtYmVyKGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIgfHwgJycpO1xyXG4gICAgc2V0Rm9ybURhdGFzZXROYW1lKGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSB8fCAnJyk7XHJcbiAgfSwgW2luaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIsIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSwgZm9ybV0pO1xyXG5cclxuICBjb25zdCBsYXlvdXQgPSB7XHJcbiAgICBsYWJlbENvbDogeyBzcGFuOiA4IH0sXHJcbiAgICB3cmFwcGVyQ29sOiB7IHNwYW46IDE2IH0sXHJcbiAgfTtcclxuXHJcbiAgcmV0dXJuIChcclxuICAgIDxkaXYgc3R5bGU9e3tqdXN0aWZ5Q29udGVudDogJ2NlbnRlcicsIHdpZHRoOiAnbWF4LWNvbnRlbnQnfX0+IFxyXG4gICAgICA8Q3VzdG9tRm9ybVxyXG4gICAgICAgIGZvcm09e2Zvcm19XHJcbiAgICAgICAgbGF5b3V0PXsnaW5saW5lJ31cclxuICAgICAgICBqdXN0aWZ5Y29udGVudD1cImNlbnRlclwiXHJcbiAgICAgICAgey4uLmxheW91dH1cclxuICAgICAgICBuYW1lPXtgc2VhcmNoX2Zvcm0ke3R5cGV9YH1cclxuICAgICAgICBjbGFzc05hbWU9XCJmaWVsZExhYmVsXCJcclxuICAgICAgICBpbml0aWFsVmFsdWVzPXt7XHJcbiAgICAgICAgICBydW5fbnVtYmVyOiBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyLFxyXG4gICAgICAgICAgZGF0YXNldF9uYW1lOiBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUsXHJcbiAgICAgICAgfX1cclxuICAgICAgICBvbkZpbmlzaD17KCkgPT4ge1xyXG4gICAgICAgICAgc2V0UnVuTnVtYmVyICYmIHNldFJ1bk51bWJlcihmb3JtX3NlYXJjaF9ydW5fbnVtYmVyKTtcclxuICAgICAgICAgIHNldERhdGFzZXROYW1lICYmIHNldERhdGFzZXROYW1lKGZvcm1fc2VhcmNoX2RhdGFzZXRfbmFtZSk7XHJcbiAgICAgICAgfX1cclxuICAgICAgPlxyXG4gICAgICAgIDxGb3JtLkl0ZW0+XHJcbiAgICAgICAgICA8UXVlc3Rpb25CdXR0b24gLz5cclxuICAgICAgICA8L0Zvcm0uSXRlbT5cclxuICAgICAgICA8U3R5bGVkRm9ybUl0ZW0gbmFtZT1cInJ1bl9udW1iZXJcIj5cclxuICAgICAgICAgIDxTdHlsZWRJbnB1dFxyXG4gICAgICAgICAgICBpZD1cInJ1bl9udW1iZXJcIlxyXG4gICAgICAgICAgICBvbkNoYW5nZT17KGU6IENoYW5nZUV2ZW50PEhUTUxJbnB1dEVsZW1lbnQ+KSA9PlxyXG4gICAgICAgICAgICAgIHNldEZvcm1SdW5OdW1iZXIoZS50YXJnZXQudmFsdWUpXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgICAgcGxhY2Vob2xkZXI9XCJFbnRlciBydW4gbnVtYmVyXCJcclxuICAgICAgICAgICAgdHlwZT1cInRleHRcIlxyXG4gICAgICAgICAgICBuYW1lPVwicnVuX251bWJlclwiXHJcbiAgICAgICAgICAgIHZhbHVlPXtkZWZhdWx0UnVuTnVtYmVyfVxyXG4gICAgICAgICAgLz5cclxuICAgICAgICA8L1N0eWxlZEZvcm1JdGVtPlxyXG4gICAgICAgIHtmdW5jdGlvbnNfY29uZmlnLm1vZGUgIT09ICdPTkxJTkUnICYmIChcclxuICAgICAgICAgIDxTdHlsZWRGb3JtSXRlbSBuYW1lPVwiZGF0YXNldF9uYW1lXCI+XHJcbiAgICAgICAgICAgIDxTdHlsZWRJbnB1dFxyXG4gICAgICAgICAgICAgIGlkPVwiZGF0YXNldF9uYW1lXCJcclxuICAgICAgICAgICAgICBwbGFjZWhvbGRlcj1cIkVudGVyIGRhdGFzZXQgbmFtZVwiXHJcbiAgICAgICAgICAgICAgb25DaGFuZ2U9eyhlOiBDaGFuZ2VFdmVudDxIVE1MSW5wdXRFbGVtZW50PikgPT5cclxuICAgICAgICAgICAgICAgIHNldEZvcm1EYXRhc2V0TmFtZShlLnRhcmdldC52YWx1ZSlcclxuICAgICAgICAgICAgICB9XHJcbiAgICAgICAgICAgICAgdHlwZT1cInRleHRcIlxyXG4gICAgICAgICAgICAgIHZhbHVlPXtkZWZhdWx0RGF0YXNldE5hbWV9XHJcbiAgICAgICAgICAgIC8+XHJcbiAgICAgICAgICA8L1N0eWxlZEZvcm1JdGVtPlxyXG4gICAgICAgICl9XHJcbiAgICAgICAgPEZvcm0uSXRlbSA+XHJcbiAgICAgICAgICA8U2VhcmNoQnV0dG9uXHJcbiAgICAgICAgICAgIG9uQ2xpY2s9eygpID0+XHJcbiAgICAgICAgICAgICAgaGFuZGxlcihmb3JtX3NlYXJjaF9ydW5fbnVtYmVyLCBmb3JtX3NlYXJjaF9kYXRhc2V0X25hbWUpXHJcbiAgICAgICAgICAgIH1cclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgPC9Gb3JtLkl0ZW0+XHJcbiAgICAgIDwvQ3VzdG9tRm9ybT5cclxuICAgIDwvZGl2PlxyXG4gICk7XHJcbn07XHJcblxyXG5leHBvcnQgZGVmYXVsdCBOYXY7XHJcbiJdLCJzb3VyY2VSb290IjoiIn0=