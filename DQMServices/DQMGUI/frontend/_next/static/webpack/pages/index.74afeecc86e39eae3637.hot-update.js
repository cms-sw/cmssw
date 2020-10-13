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



var _this = undefined,
    _jsxFileName = "/mnt/c/Users/ernes/Desktop/test/dqmgui_frontend/components/Nav.tsx",
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_2___default.a.createElement;






var Nav = function Nav(_ref) {
  _s();

  var initial_search_run_number = _ref.initial_search_run_number,
      initial_search_dataset_name = _ref.initial_search_dataset_name,
      initial_search_lumisection = _ref.initial_search_lumisection,
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
      setFormDatasetName = _useState2[1];

  var _useState3 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initial_search_lumisection || ''),
      form_search_lumisection = _useState3[0],
      setFormLumisection = _useState3[1]; // We have to wait for changin initial_search_run_number and initial_search_dataset_name coming from query, because the first render they are undefined and therefore the initialValues doesn't grab them


  Object(react__WEBPACK_IMPORTED_MODULE_2__["useEffect"])(function () {
    form.resetFields();
    setFormRunNumber(initial_search_run_number || '');
    setFormDatasetName(initial_search_dataset_name || '');
    setFormLumisection(initial_search_lumisection || '');
  }, [initial_search_run_number, initial_search_dataset_name, initial_search_lumisection, form]);
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
      lineNumber: 60,
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
      lineNumber: 61,
      columnNumber: 7
    }
  }), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({}, tailLayout, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 78,
      columnNumber: 9
    }
  }), __jsx(_helpButton__WEBPACK_IMPORTED_MODULE_6__["QuestionButton"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 79,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "run_number",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 81,
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
      lineNumber: 82,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "dataset_name",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 93,
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
      lineNumber: 94,
      columnNumber: 11
    }
  })), _config_config__WEBPACK_IMPORTED_MODULE_7__["functions_config"].new_back_end.lumisections_on && __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "lumisection",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 106,
      columnNumber: 11
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "lumisection",
    placeholder: "Enter lumisection",
    onChange: function onChange(e) {
      return setFormLumisection(e.target.value);
    },
    type: "text",
    value: form_search_lumisection,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 107,
      columnNumber: 13
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({}, tailLayout, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 118,
      columnNumber: 9
    }
  }), __jsx(_searchButton__WEBPACK_IMPORTED_MODULE_5__["SearchButton"], {
    onClick: function onClick() {
      return handler(form_search_run_number, form_search_dataset_name, form_search_lumisection);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 119,
      columnNumber: 11
    }
  }))));
};

_s(Nav, "fe/Qdjxsdj3yTEyDYMx5Agsmx2g=", false, function () {
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9OYXYudHN4Il0sIm5hbWVzIjpbIk5hdiIsImluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIiLCJpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUiLCJpbml0aWFsX3NlYXJjaF9sdW1pc2VjdGlvbiIsInNldFJ1bk51bWJlciIsInNldERhdGFzZXROYW1lIiwiaGFuZGxlciIsInR5cGUiLCJkZWZhdWx0UnVuTnVtYmVyIiwiZGVmYXVsdERhdGFzZXROYW1lIiwiRm9ybSIsInVzZUZvcm0iLCJmb3JtIiwidXNlU3RhdGUiLCJmb3JtX3NlYXJjaF9ydW5fbnVtYmVyIiwic2V0Rm9ybVJ1bk51bWJlciIsImZvcm1fc2VhcmNoX2RhdGFzZXRfbmFtZSIsInNldEZvcm1EYXRhc2V0TmFtZSIsImZvcm1fc2VhcmNoX2x1bWlzZWN0aW9uIiwic2V0Rm9ybUx1bWlzZWN0aW9uIiwidXNlRWZmZWN0IiwicmVzZXRGaWVsZHMiLCJsYXlvdXQiLCJsYWJlbENvbCIsInNwYW4iLCJ3cmFwcGVyQ29sIiwidGFpbExheW91dCIsIm9mZnNldCIsInJ1bl9udW1iZXIiLCJkYXRhc2V0X25hbWUiLCJlIiwidGFyZ2V0IiwidmFsdWUiLCJmdW5jdGlvbnNfY29uZmlnIiwibmV3X2JhY2tfZW5kIiwibHVtaXNlY3Rpb25zX29uIl0sIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztBQUFBO0FBQ0E7QUFFQTtBQUNBO0FBQ0E7QUFDQTtBQWNPLElBQU1BLEdBQUcsR0FBRyxTQUFOQSxHQUFNLE9BVUg7QUFBQTs7QUFBQSxNQVRkQyx5QkFTYyxRQVRkQSx5QkFTYztBQUFBLE1BUmRDLDJCQVFjLFFBUmRBLDJCQVFjO0FBQUEsTUFQZEMsMEJBT2MsUUFQZEEsMEJBT2M7QUFBQSxNQU5kQyxZQU1jLFFBTmRBLFlBTWM7QUFBQSxNQUxkQyxjQUtjLFFBTGRBLGNBS2M7QUFBQSxNQUpkQyxPQUljLFFBSmRBLE9BSWM7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OztLQVhUWCxHO0FBNkdFQSxrRUFBZiIsImZpbGUiOiJzdGF0aWMvd2VicGFjay9wYWdlcy9pbmRleC43NGFmZWVjYzg2ZTM5ZWFlMzYzNy5ob3QtdXBkYXRlLmpzIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IFJlYWN0LCB7IENoYW5nZUV2ZW50LCBEaXNwYXRjaCwgdXNlRWZmZWN0LCB1c2VTdGF0ZSB9IGZyb20gJ3JlYWN0JztcbmltcG9ydCB7IEZvcm0gfSBmcm9tICdhbnRkJztcblxuaW1wb3J0IHsgU3R5bGVkRm9ybUl0ZW0sIFN0eWxlZElucHV0LCBDdXN0b21Gb3JtIH0gZnJvbSAnLi9zdHlsZWRDb21wb25lbnRzJztcbmltcG9ydCB7IFNlYXJjaEJ1dHRvbiB9IGZyb20gJy4vc2VhcmNoQnV0dG9uJztcbmltcG9ydCB7IFF1ZXN0aW9uQnV0dG9uIH0gZnJvbSAnLi9oZWxwQnV0dG9uJztcbmltcG9ydCB7IGZ1bmN0aW9uc19jb25maWcgfSBmcm9tICcuLi9jb25maWcvY29uZmlnJztcblxuaW50ZXJmYWNlIE5hdlByb3BzIHtcbiAgc2V0UnVuTnVtYmVyPzogRGlzcGF0Y2g8YW55PjtcbiAgc2V0RGF0YXNldE5hbWU/OiBEaXNwYXRjaDxhbnk+O1xuICBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyPzogc3RyaW5nO1xuICBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWU/OiBzdHJpbmc7XG4gIGluaXRpYWxfc2VhcmNoX2x1bWlzZWN0aW9uPzogc3RyaW5nO1xuICBoYW5kbGVyKHNlYXJjaF9ieV9ydW5fbnVtYmVyOiBzdHJpbmcsIHNlYXJjaF9ieV9kYXRhc2V0X25hbWU6IHN0cmluZyk6IHZvaWQ7XG4gIHR5cGU6IHN0cmluZztcbiAgZGVmYXVsdFJ1bk51bWJlcj86IHVuZGVmaW5lZCB8IHN0cmluZztcbiAgZGVmYXVsdERhdGFzZXROYW1lPzogc3RyaW5nIHwgdW5kZWZpbmVkO1xufVxuXG5leHBvcnQgY29uc3QgTmF2ID0gKHtcbiAgaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlcixcbiAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lLFxuICBpbml0aWFsX3NlYXJjaF9sdW1pc2VjdGlvbixcbiAgc2V0UnVuTnVtYmVyLFxuICBzZXREYXRhc2V0TmFtZSxcbiAgaGFuZGxlcixcbiAgdHlwZSxcbiAgZGVmYXVsdFJ1bk51bWJlcixcbiAgZGVmYXVsdERhdGFzZXROYW1lLFxufTogTmF2UHJvcHMpID0+IHtcbiAgY29uc3QgW2Zvcm1dID0gRm9ybS51c2VGb3JtKCk7XG4gIGNvbnN0IFtmb3JtX3NlYXJjaF9ydW5fbnVtYmVyLCBzZXRGb3JtUnVuTnVtYmVyXSA9IHVzZVN0YXRlKFxuICAgIGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIgfHwgJydcbiAgKTtcbiAgY29uc3QgW2Zvcm1fc2VhcmNoX2RhdGFzZXRfbmFtZSwgc2V0Rm9ybURhdGFzZXROYW1lXSA9IHVzZVN0YXRlKFxuICAgIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSB8fCAnJ1xuICApO1xuICBjb25zdCBbZm9ybV9zZWFyY2hfbHVtaXNlY3Rpb24sIHNldEZvcm1MdW1pc2VjdGlvbl0gPSB1c2VTdGF0ZShcbiAgICBpbml0aWFsX3NlYXJjaF9sdW1pc2VjdGlvbiB8fCAnJ1xuICApO1xuXG4gIC8vIFdlIGhhdmUgdG8gd2FpdCBmb3IgY2hhbmdpbiBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyIGFuZCBpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUgY29taW5nIGZyb20gcXVlcnksIGJlY2F1c2UgdGhlIGZpcnN0IHJlbmRlciB0aGV5IGFyZSB1bmRlZmluZWQgYW5kIHRoZXJlZm9yZSB0aGUgaW5pdGlhbFZhbHVlcyBkb2Vzbid0IGdyYWIgdGhlbVxuICB1c2VFZmZlY3QoKCkgPT4ge1xuICAgIGZvcm0ucmVzZXRGaWVsZHMoKTtcbiAgICBzZXRGb3JtUnVuTnVtYmVyKGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIgfHwgJycpO1xuICAgIHNldEZvcm1EYXRhc2V0TmFtZShpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUgfHwgJycpO1xuICAgIHNldEZvcm1MdW1pc2VjdGlvbihpbml0aWFsX3NlYXJjaF9sdW1pc2VjdGlvbiB8fCAnJylcbiAgfSwgW2luaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIsIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSwgaW5pdGlhbF9zZWFyY2hfbHVtaXNlY3Rpb24sZm9ybV0pO1xuXG4gIGNvbnN0IGxheW91dCA9IHtcbiAgICBsYWJlbENvbDogeyBzcGFuOiA4IH0sXG4gICAgd3JhcHBlckNvbDogeyBzcGFuOiAxNiB9LFxuICB9O1xuICBjb25zdCB0YWlsTGF5b3V0ID0ge1xuICAgIHdyYXBwZXJDb2w6IHsgb2Zmc2V0OiAwLCBzcGFuOiA0IH0sXG4gIH07XG5cbiAgcmV0dXJuIChcbiAgICA8ZGl2PlxuICAgICAgPEN1c3RvbUZvcm1cbiAgICAgICAgZm9ybT17Zm9ybX1cbiAgICAgICAgbGF5b3V0PXsnaW5saW5lJ31cbiAgICAgICAganVzdGlmeWNvbnRlbnQ9XCJjZW50ZXJcIlxuICAgICAgICB3aWR0aD1cIm1heC1jb250ZW50XCJcbiAgICAgICAgey4uLmxheW91dH1cbiAgICAgICAgbmFtZT17YHNlYXJjaF9mb3JtJHt0eXBlfWB9XG4gICAgICAgIGNsYXNzTmFtZT1cImZpZWxkTGFiZWxcIlxuICAgICAgICBpbml0aWFsVmFsdWVzPXt7XG4gICAgICAgICAgcnVuX251bWJlcjogaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlcixcbiAgICAgICAgICBkYXRhc2V0X25hbWU6IGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSxcbiAgICAgICAgfX1cbiAgICAgICAgb25GaW5pc2g9eygpID0+IHtcbiAgICAgICAgICBzZXRSdW5OdW1iZXIgJiYgc2V0UnVuTnVtYmVyKGZvcm1fc2VhcmNoX3J1bl9udW1iZXIpO1xuICAgICAgICAgIHNldERhdGFzZXROYW1lICYmIHNldERhdGFzZXROYW1lKGZvcm1fc2VhcmNoX2RhdGFzZXRfbmFtZSk7XG4gICAgICAgIH19XG4gICAgICA+XG4gICAgICAgIDxGb3JtLkl0ZW0gey4uLnRhaWxMYXlvdXR9PlxuICAgICAgICAgIDxRdWVzdGlvbkJ1dHRvbiAvPlxuICAgICAgICA8L0Zvcm0uSXRlbT5cbiAgICAgICAgPFN0eWxlZEZvcm1JdGVtIG5hbWU9XCJydW5fbnVtYmVyXCI+XG4gICAgICAgICAgPFN0eWxlZElucHV0XG4gICAgICAgICAgICBpZD1cInJ1bl9udW1iZXJcIlxuICAgICAgICAgICAgb25DaGFuZ2U9eyhlOiBDaGFuZ2VFdmVudDxIVE1MSW5wdXRFbGVtZW50PikgPT5cbiAgICAgICAgICAgICAgc2V0Rm9ybVJ1bk51bWJlcihlLnRhcmdldC52YWx1ZSlcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIHBsYWNlaG9sZGVyPVwiRW50ZXIgcnVuIG51bWJlclwiXG4gICAgICAgICAgICB0eXBlPVwidGV4dFwiXG4gICAgICAgICAgICBuYW1lPVwicnVuX251bWJlclwiXG4gICAgICAgICAgICB2YWx1ZT17ZGVmYXVsdFJ1bk51bWJlcn1cbiAgICAgICAgICAvPlxuICAgICAgICA8L1N0eWxlZEZvcm1JdGVtPlxuICAgICAgICA8U3R5bGVkRm9ybUl0ZW0gbmFtZT1cImRhdGFzZXRfbmFtZVwiPlxuICAgICAgICAgIDxTdHlsZWRJbnB1dFxuICAgICAgICAgICAgaWQ9XCJkYXRhc2V0X25hbWVcIlxuICAgICAgICAgICAgcGxhY2Vob2xkZXI9XCJFbnRlciBkYXRhc2V0IG5hbWVcIlxuICAgICAgICAgICAgb25DaGFuZ2U9eyhlOiBDaGFuZ2VFdmVudDxIVE1MSW5wdXRFbGVtZW50PikgPT5cbiAgICAgICAgICAgICAgc2V0Rm9ybURhdGFzZXROYW1lKGUudGFyZ2V0LnZhbHVlKVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgdHlwZT1cInRleHRcIlxuICAgICAgICAgICAgdmFsdWU9e2RlZmF1bHREYXRhc2V0TmFtZX1cbiAgICAgICAgICAvPlxuICAgICAgICA8L1N0eWxlZEZvcm1JdGVtPlxuICAgICAgICB7XG4gICAgICAgICAgZnVuY3Rpb25zX2NvbmZpZy5uZXdfYmFja19lbmQubHVtaXNlY3Rpb25zX29uICYmXG4gICAgICAgICAgPFN0eWxlZEZvcm1JdGVtIG5hbWU9XCJsdW1pc2VjdGlvblwiPlxuICAgICAgICAgICAgPFN0eWxlZElucHV0XG4gICAgICAgICAgICAgIGlkPVwibHVtaXNlY3Rpb25cIlxuICAgICAgICAgICAgICBwbGFjZWhvbGRlcj1cIkVudGVyIGx1bWlzZWN0aW9uXCJcbiAgICAgICAgICAgICAgb25DaGFuZ2U9eyhlOiBDaGFuZ2VFdmVudDxIVE1MSW5wdXRFbGVtZW50PikgPT5cbiAgICAgICAgICAgICAgICBzZXRGb3JtTHVtaXNlY3Rpb24oZS50YXJnZXQudmFsdWUpXG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgdHlwZT1cInRleHRcIlxuICAgICAgICAgICAgICB2YWx1ZT17Zm9ybV9zZWFyY2hfbHVtaXNlY3Rpb259XG4gICAgICAgICAgICAvPlxuICAgICAgICAgIDwvU3R5bGVkRm9ybUl0ZW0+XG4gICAgICAgIH1cbiAgICAgICAgPEZvcm0uSXRlbSB7Li4udGFpbExheW91dH0+XG4gICAgICAgICAgPFNlYXJjaEJ1dHRvblxuICAgICAgICAgICAgb25DbGljaz17KCkgPT5cbiAgICAgICAgICAgICAgaGFuZGxlcihmb3JtX3NlYXJjaF9ydW5fbnVtYmVyLCBmb3JtX3NlYXJjaF9kYXRhc2V0X25hbWUsIGZvcm1fc2VhcmNoX2x1bWlzZWN0aW9uKVxuICAgICAgICAgICAgfVxuICAgICAgICAgIC8+XG4gICAgICAgIDwvRm9ybS5JdGVtPlxuICAgICAgPC9DdXN0b21Gb3JtPlxuICAgIDwvZGl2PlxuICApO1xufTtcblxuZXhwb3J0IGRlZmF1bHQgTmF2O1xuIl0sInNvdXJjZVJvb3QiOiIifQ==